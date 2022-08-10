CDF   �   
      time       bnds      lon       lat          7   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       ,CanESM5 (2019): 
aerosol: interactive
atmos: CanAM5 (T63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa)
atmosChem: specified oxidants for aerosols
land: CLASS3.6/CTEM1.2
landIce: specified ice sheets
ocean: NEMO3.4.1 (ORCA1 tripolar grid, 1 deg with refinement to 1/3 deg within 20 degrees of the equator; 361 x 290 longitude/latitude; 45 vertical levels; top grid cell 0-6.19 m)
ocnBgchem: Canadian Model of Ocean Carbon (CMOC); NPZD ecosystem with OMIP prescribed carbonate chemistry
seaIce: LIM2   institution       wCanadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada, Victoria, BC V8P 5C2, Canada    CCCma_model_hash      (3dedf95315d603326fde4f5340dc0519d80d10c0   CCCma_parent_runid        
rc3-pictrl     CCCma_pycmor_hash         (33c30511acc319a98240633965a04ca99c26427e   CCCma_runid       rc3.1-his01    YMDH_branch_time_in_child         1850:01:01:00      YMDH_branch_time_in_parent        5201:01:01:00      activity_id       CMIP   branch_method         Spin-up documentation      branch_time_in_child                 branch_time_in_parent         A2��       contact       %ec.cccma.info-info.ccmac.ec@canada.ca      creation_date         2019-04-30T17:32:13Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.CCCma.CanESM5.historical.none.r1i1p1f1    grid      kT63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa    
grid_label        gn     history      	�Wed Aug 10 15:17:36 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/CanESM5_r1i1p1f1/rsdt_CanESM5_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/CanESM5_r1i1p1f1/CMIP6.ScenarioMIP.CCCma.CanESM5.ssp245.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_ssp245_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/CanESM5_r1i1p1f1/rsdt_CanESM5_r1i1p1f1_ssp245.mergetime.nc
Wed Aug 10 15:17:35 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/CanESM5_r1i1p1f1/rsdt_CanESM5_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 07:27:20 2022: cdo -O -s -selname,rsdt -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 07:27:16 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsdt -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsdt/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.rsdt.gn.v20190429/rsdt_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-04-30T17:32:13Z ;rewrote data to be consistent with CMIP for variable rsdt found in table Amon.;
Output from $runid   initialization_index            institution_id        CCCma      mip_era       CMIP6      nominal_resolution        500 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      CanESM5    parent_time_units         days since 1850-01-01 0:0:0.0      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      
references        mGeophysical Model Development Special issue on CanESM5 (https://www.geosci-model-dev.net/special_issues.html)      	source_id         CanESM5    source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(20 February 2019) MD5:374fbe5a2bcca535c40f7f23da271e49      title         !CanESM5 output prepared for CMIP6      tracking_id       1hdl:21.14100/1ab756cd-1928-40f9-82dc-a0e58a183ed8      variable_id       rsdt   variant_label         r1i1p1f1   version       	v20190429      license      �CMIP6 model data produced by The Government of Canada (Canadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada) is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               4   	time_bnds                                 <   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               $   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               ,   rsdt                   
   standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      original_name         FSO    cell_measures         area: areacella    history       �2019-04-30T17:32:13Z altered by CMOR: Reordered dimensions, original order: lat lon time. 2019-04-30T17:32:13Z altered by CMOR: replaced missing value flag (1e+38) with standard missing value (1e+20).            L                Aq���   Aq��P   Aq�P   C�"Aq�6�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C��Aq��   Aq��P   Aq�dP   C��Aq���   Aq�dP   Aq��P   C�7Aq���   Aq��P   Aq�FP   C� Aq�k�   Aq�FP   Aq��P   C�SAq���   Aq��P   Aq�(P   C��Aq�M�   Aq�(P   Aq��P   C�^Aq���   Aq��P   Aq�
P   C��Aq�/�   Aq�
P   Aq�{P   C�# Aq���   Aq�{P   Aq��P   C�"�Aq��   Aq��P   Aq�]P   C��AqĂ�   Aq�]P   Aq��P   C��Aq���   Aq��P   Aq�?P   C��Aq�d�   Aq�?P   Aq˰P   C��Aq���   Aq˰P   Aq�!P   C�Aq�F�   Aq�!P   AqВP   C�!Aqз�   AqВP   Aq�P   C��Aq�(�   Aq�P   Aq�tP   C�RAqՙ�   Aq�tP   Aq��P   C�<Aq�
�   Aq��P   Aq�VP   C�"aAq�{�   Aq�VP   Aq��P   C� �Aq���   Aq��P   Aq�8P   C�QAq�]�   Aq�8P   Aq�P   C��Aq���   Aq�P   Aq�P   C�<Aq�?�   Aq�P   Aq�P   C�_Aq��   Aq�P   Aq��P   C�Aq�!�   Aq��P   Aq�mP   C��Aq��   Aq�mP   Aq��P   C��Aq��   Aq��P   Aq�OP   C��Aq�t�   Aq�OP   Aq��P   C��Aq���   Aq��P   Aq�1P   C��Aq�V�   Aq�1P   Aq��P   C��Aq���   Aq��P   Aq�P   C�!Aq�8�   Aq�P   Aq��P   C�xAq���   Aq��P   Aq��P   C�BAq��   Aq��P   ArfP   C�rAr��   ArfP   Ar�P   C��Ar��   Ar�P   ArHP   C��Arm�   ArHP   Ar�P   C�SAr��   Ar�P   Ar*P   C�,ArO�   Ar*P   Ar�P   C�!Ar��   Ar�P   ArP   C�&WAr1�   ArP   Ar}P   C�)�Ar��   Ar}P   Ar�P   C�&�Ar�   Ar�P   Ar_P   C� uAr��   Ar_P   Ar�P   C�\Ar��   Ar�P   ArAP   C��Arf�   ArAP   Ar�P   C�5Ar��   Ar�P   Ar!#P   C��Ar!H�   Ar!#P   Ar#�P   C�WAr#��   Ar#�P   Ar&P   C��Ar&*�   Ar&P   Ar(vP   C�tAr(��   Ar(vP   Ar*�P   C�!Ar+�   Ar*�P   Ar-XP   C�GAr-}�   Ar-XP   Ar/�P   C�$xAr/��   Ar/�P   Ar2:P   C� Ar2_�   Ar2:P   Ar4�P   C�"IAr4��   Ar4�P   Ar7P   C�xAr7A�   Ar7P   Ar9�P   C��Ar9��   Ar9�P   Ar;�P   C�LAr<#�   Ar;�P   Ar>oP   C�LAr>��   Ar>oP   Ar@�P   C��ArA�   Ar@�P   ArCQP   C�_ArCv�   ArCQP   ArE�P   C�!�ArE��   ArE�P   ArH3P   C�*1ArHX�   ArH3P   ArJ�P   C�/�ArJ��   ArJ�P   ArMP   C�0�ArM:�   ArMP   ArO�P   C�(�ArO��   ArO�P   ArQ�P   C�!�ArR�   ArQ�P   ArThP   C�#ArT��   ArThP   ArV�P   C�xArV��   ArV�P   ArYJP   C��ArYo�   ArYJP   Ar[�P   C��Ar[��   Ar[�P   Ar^,P   C��Ar^Q�   Ar^,P   Ar`�P   C�"�Ar`��   Ar`�P   ArcP   C�*]Arc3�   ArcP   AreP   C�'TAre��   AreP   Arg�P   C�&�Arh�   Arg�P   ArjaP   C�#�Arj��   ArjaP   Arl�P   C� UArl��   Arl�P   AroCP   C�
Aroh�   AroCP   Arq�P   C�
Arq��   Arq�P   Art%P   C��ArtJ�   Art%P   Arv�P   C�"EArv��   Arv�P   AryP   C�.�Ary,�   AryP   Ar{xP   C�1Ar{��   Ar{xP   Ar}�P   C�.�Ar~�   Ar}�P   Ar�ZP   C�-�Ar��   Ar�ZP   Ar��P   C�)�Ar���   Ar��P   Ar�<P   C�%�Ar�a�   Ar�<P   Ar��P   C�!�Ar���   Ar��P   Ar�P   C��Ar�C�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar� P   C�%Ar�%�   Ar� P   Ar�qP   C�*qAr���   Ar�qP   Ar��P   C�3kAr��   Ar��P   Ar�SP   C�:�Ar�x�   Ar�SP   Ar��P   C�49Ar���   Ar��P   Ar�5P   C�.�Ar�Z�   Ar�5P   Ar��P   C�#Ar���   Ar��P   Ar�P   C�"�Ar�<�   Ar�P   Ar��P   C�Ar���   Ar��P   Ar��P   C��Ar��   Ar��P   Ar�jP   C�$tAr���   Ar�jP   Ar��P   C�3Ar� �   Ar��P   Ar�LP   C�A9Ar�q�   Ar�LP   Ar��P   C�DAr���   Ar��P   Ar�.P   C�<@Ar�S�   Ar�.P   Ar��P   C�6�Ar���   Ar��P   Ar�P   C�,]Ar�5�   Ar�P   Ar��P   C�#+Ar���   Ar��P   Ar��P   C� &Ar��   Ar��P   Ar�cP   C�Ar���   Ar�cP   Ar��P   C�qAr���   Ar��P   Ar�EP   C�$�Ar�j�   Ar�EP   ArĶP   C�-+Ar���   ArĶP   Ar�'P   C�1QAr�L�   Ar�'P   ArɘP   C�1�Arɽ�   ArɘP   Ar�	P   C�28Ar�.�   Ar�	P   Ar�zP   C�,�ArΟ�   Ar�zP   Ar��P   C�+{Ar��   Ar��P   Ar�\P   C�$�ArӁ�   Ar�\P   Ar��P   C� �Ar���   Ar��P   Ar�>P   C�!Ar�c�   Ar�>P   ArگP   C� Ar���   ArگP   Ar� P   C�%�Ar�E�   Ar� P   ArߑP   C�3>Ar߶�   ArߑP   Ar�P   C�9fAr�'�   Ar�P   Ar�sP   C�:Ar��   Ar�sP   Ar��P   C�:�Ar�	�   Ar��P   Ar�UP   C�0fAr�z�   Ar�UP   Ar��P   C�0&Ar���   Ar��P   Ar�7P   C�$Ar�\�   Ar�7P   Ar�P   C�
Ar���   Ar�P   Ar�P   C��Ar�>�   Ar�P   Ar��P   C�!yAr���   Ar��P   Ar��P   C�,zAr� �   Ar��P   Ar�lP   C�:JAr���   Ar�lP   Ar��P   C�:�Ar��   Ar��P   Ar�NP   C�9,Ar�s�   Ar�NP   As�P   C�3As��   As�P   As0P   C�)9AsU�   As0P   As�P   C�!As��   As�P   As	P   C��As	7�   As	P   As�P   C�_As��   As�P   As�P   C� As�   As�P   AseP   C�*oAs��   AseP   As�P   C�4_As��   As�P   AsGP   C�9)Asl�   AsGP   As�P   C�8As��   As�P   As)P   C�9�AsN�   As)P   As�P   C�+�As��   As�P   AsP   C�#�As0�   AsP   As!|P   C��As!��   As!|P   As#�P   C�&As$�   As#�P   As&^P   C��As&��   As&^P   As(�P   C�iAs(��   As(�P   As+@P   C�As+e�   As+@P   As-�P   C�pAs-��   As-�P   As0"P   C�$SAs0G�   As0"P   As2�P   C�*7As2��   As2�P   As5P   C�+(As5)�   As5P   As7uP   C�-�As7��   As7uP   As9�P   C�(&As:�   As9�P   As<WP   C�?As<|�   As<WP   As>�P   C��As>��   As>�P   AsA9P   C�AsA^�   AsA9P   AsC�P   C�AsC��   AsC�P   AsFP   C�7AsF@�   AsFP   AsH�P   C�#)AsH��   AsH�P   AsJ�P   C�,�AsK"�   AsJ�P   AsMnP   C�3�AsM��   AsMnP   AsO�P   C�5mAsP�   AsO�P   AsRPP   C�0dAsRu�   AsRPP   AsT�P   C�*�AsT��   AsT�P   AsW2P   C�$�AsWW�   AsW2P   AsY�P   C�!AsY��   AsY�P   As\P   C�As\9�   As\P   As^�P   C��As^��   As^�P   As`�P   C�Asa�   As`�P   AscgP   C�dAsc��   AscgP   Ase�P   C��Ase��   Ase�P   AshIP   C�'�Ashn�   AshIP   Asj�P   C�(�Asj��   Asj�P   Asm+P   C�.rAsmP�   Asm+P   Aso�P   C�(�Aso��   Aso�P   AsrP   C�(�Asr2�   AsrP   Ast~P   C�#�Ast��   Ast~P   Asv�P   C�Asw�   Asv�P   Asy`P   C��Asy��   Asy`P   As{�P   C�;As{��   As{�P   As~BP   C��As~g�   As~BP   As��P   C�VAs���   As��P   As�$P   C�"�As�I�   As�$P   As��P   C�)5As���   As��P   As�P   C�0<As�+�   As�P   As�wP   C�1As���   As�wP   As��P   C�+}As��   As��P   As�YP   C�#�As�~�   As�YP   As��P   C��As���   As��P   As�;P   C��As�`�   As�;P   As��P   C�LAs���   As��P   As�P   C�As�B�   As�P   As��P   C�NAs���   As��P   As��P   C�$rAs�$�   As��P   As�pP   C�$�As���   As�pP   As��P   C�%�As��   As��P   As�RP   C�'QAs�w�   As�RP   As��P   C�"�As���   As��P   As�4P   C�As�Y�   As�4P   As��P   C��As���   As��P   As�P   C�RAs�;�   As�P   As��P   C��As���   As��P   As��P   C��As��   As��P   As�iP   C�qAs���   As�iP   As��P   C�&As���   As��P   As�KP   C�)�As�p�   As�KP   As��P   C�)As���   As��P   As�-P   C�$�As�R�   As�-P   AsP   C�^As���   AsP   As�P   C�mAs�4�   As�P   AsǀP   C��Asǥ�   AsǀP   As��P   C��As��   As��P   As�bP   C��Aṡ�   As�bP   As��P   C�vAs���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C��As���   AsӵP   As�&P   C��As�K�   As�&P   AsؗP   C�#OAsؼ�   AsؗP   As�P   C�#As�-�   As�P   As�yP   C�#�Asݞ�   As�yP   As��P   C� �As��   As��P   As�[P   C�uAs��   As�[P   As��P   C��As���   As��P   As�=P   C�oAs�b�   As�=P   As�P   C��As���   As�P   As�P   C�KAs�D�   As�P   As�P   C��As��   As�P   As�P   C� �As�&�   As�P   As�rP   C�%FAs��   As�rP   As��P   C�,/As��   As��P   As�TP   C�*�As�y�   As�TP   As��P   C�%mAs���   As��P   As�6P   C��As�[�   As�6P   As��P   C��As���   As��P   AtP   C��At=�   AtP   At�P   C��At��   At�P   At�P   C��At�   At�P   At	kP   C�