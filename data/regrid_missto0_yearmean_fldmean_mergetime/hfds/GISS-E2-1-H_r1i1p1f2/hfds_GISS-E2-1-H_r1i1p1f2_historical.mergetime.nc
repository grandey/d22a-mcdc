CDF   �   
      time       bnds      lon       lat          2   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �GISS-E2.1H (2016): 
aerosol: Varies with physics-version (p==1 none, p==3 OMA, p==4 TOMAS, p==5 MATRIX)
atmos: GISS-E2.1 (2.5x2 degree; 144 x 90 longitude/latitude; 40 levels; top level 0.1 hPa)
atmosChem: Varies with physics-version (p==1 Non-interactive, p>1 GPUCCINI)
land: GISS LSM
landIce: none
ocean: HYCOM Ocean (~1 degree tripolar grid; 360 x 180 latitude/longitude; 26 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: GISS SI    institution       <Goddard Institute for Space Studies, New York, NY 10025, USA   activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    contact        Kenneth Lo (cdkkl@giss.nasa.gov)   creation_date         2019-11-30T17:49:53Z   data_specs_version        01.00.23   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Shttps://furtherinfo.es-doc.org/CMIP6.NASA-GISS.GISS-E2-1-H.historical.none.r1i1p1f2    grid      -atmospheric grid: 144x90, ocean grid: 360x180      
grid_label        gn     history      
�Wed Aug 10 15:21:21 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_185001-190012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_190101-195012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_195101-200012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_200101-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/GISS-E2-1-H_r1i1p1f2/hfds_GISS-E2-1-H_r1i1p1f2_historical.mergetime.nc
Mon Apr 11 09:14:03 2022: cdo -O -s -selname,hfds -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_185001-190012.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_185001-190012.bic_missto0.yearmean.fldmean.nc
Mon Apr 11 09:13:59 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,hfds -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_185001-190012.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_185001-190012.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/GISS-E2-1-H_r1i1p1f2/CMIP6.CMIP.NASA-GISS.GISS-E2-1-H.historical.r1i1p1f2.Omon.hfds.gn.v20191003/hfds_Omon_GISS-E2-1-H_historical_r1i1p1f2_gn_185001-190012.bic_missto0.yearmean.nc
2019-11-30T17:49:53Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.     initialization_index            institution_id        	NASA-GISS      mip_era       CMIP6      model_id      	Eh213f10a      nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     parent_mip_era        CMIP6      parent_source_id      GISS-E2-1-H    parent_time_units         days since 4000-1-1    parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         ocean      
references        'https://data.giss.nasa.gov/modelE/cmip6    	source_id         GISS-E2-1-H    source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(21 March 2018) MD5:c93735846d66458966fc81f390b2d714     title         %GISS-E2-1-H output prepared for CMIP6      tracking_id       1hdl:21.14100/7a4b1650-99b4-49dd-afad-74ee51b200fd      variable_id       hfds   variant_label         r1i1p1f2   license      cCMIP6 model data produced by NASA Goddard Institute for Space Studies is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.3.2      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               p   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               x   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-30T17:49:53Z altered by CMOR: replaced missing value flag (-1e+30) with standard missing value (1e+20).             �                Aq���   Aq��P   Aq�P   ?��2Aq�6�   Aq�P   Aq��P   ?�Y�Aq���   Aq��P   Aq��P   ?�>�Aq��   Aq��P   Aq�dP   ?�JUAq���   Aq�dP   Aq��P   ?��/Aq���   Aq��P   Aq�FP   >�ڥAq�k�   Aq�FP   Aq��P   ?���Aq���   Aq��P   Aq�(P   ?�
Aq�M�   Aq�(P   Aq��P   ?�M�Aq���   Aq��P   Aq�
P   ?^lrAq�/�   Aq�
P   Aq�{P   ?5rZAq���   Aq�{P   Aq��P   ?g� Aq��   Aq��P   Aq�]P   ?���AqĂ�   Aq�]P   Aq��P   ?���Aq���   Aq��P   Aq�?P   ?���Aq�d�   Aq�?P   Aq˰P   ?�0EAq���   Aq˰P   Aq�!P   ?�,=Aq�F�   Aq�!P   AqВP   ?��Aqз�   AqВP   Aq�P   ?��&Aq�(�   Aq�P   Aq�tP   ?� �Aqՙ�   Aq�tP   Aq��P   ?E�rAq�
�   Aq��P   Aq�VP   ?�=Aq�{�   Aq�VP   Aq��P   ?��MAq���   Aq��P   Aq�8P   ?	��Aq�]�   Aq�8P   Aq�P   ?r��Aq���   Aq�P   Aq�P   ?�Z4Aq�?�   Aq�P   Aq�P   ?��YAq��   Aq�P   Aq��P   ?���Aq�!�   Aq��P   Aq�mP   ?HfAq��   Aq�mP   Aq��P   ?��Aq��   Aq��P   Aq�OP   ?��Aq�t�   Aq�OP   Aq��P   ?�~Aq���   Aq��P   Aq�1P   ?&\�Aq�V�   Aq�1P   Aq��P   >��Aq���   Aq��P   Aq�P   >^��Aq�8�   Aq�P   Aq��P   ?���Aq���   Aq��P   Aq��P   ?���Aq��   Aq��P   ArfP   ?�eAr��   ArfP   Ar�P   ?�W�Ar��   Ar�P   ArHP   ?���Arm�   ArHP   Ar�P   ?3T?Ar��   Ar�P   Ar*P   ?��3ArO�   Ar*P   Ar�P   ?�؋Ar��   Ar�P   ArP   ?�D�Ar1�   ArP   Ar}P   ?���Ar��   Ar}P   Ar�P   ?�; Ar�   Ar�P   Ar_P   ?�׊Ar��   Ar_P   Ar�P   ?t��Ar��   Ar�P   ArAP   ?�3�Arf�   ArAP   Ar�P   ?�o�Ar��   Ar�P   Ar!#P   ?>C�Ar!H�   Ar!#P   Ar#�P   ?�<WAr#��   Ar#�P   Ar&P   ?o��Ar&*�   Ar&P   Ar(vP   ?YZ:Ar(��   Ar(vP   Ar*�P   ?���Ar+�   Ar*�P   Ar-XP   ?�t�Ar-}�   Ar-XP   Ar/�P   ?�J{Ar/��   Ar/�P   Ar2:P   ?Ν�Ar2_�   Ar2:P   Ar4�P   ?{4�Ar4��   Ar4�P   Ar7P   ?��KAr7A�   Ar7P   Ar9�P   ?�#�Ar9��   Ar9�P   Ar;�P   ?��Ar<#�   Ar;�P   Ar>oP   ?[�Ar>��   Ar>oP   Ar@�P   ?neFArA�   Ar@�P   ArCQP   ?uu�ArCv�   ArCQP   ArE�P   ?�SxArE��   ArE�P   ArH3P   ?���ArHX�   ArH3P   ArJ�P   ?���ArJ��   ArJ�P   ArMP   ?�nfArM:�   ArMP   ArO�P   ?��ArO��   ArO�P   ArQ�P   ?�
�ArR�   ArQ�P   ArThP   ?��ArT��   ArThP   ArV�P   ?:��ArV��   ArV�P   ArYJP   ?˳�ArYo�   ArYJP   Ar[�P   ?Ӥ�Ar[��   Ar[�P   Ar^,P   ?�GAr^Q�   Ar^,P   Ar`�P   ?�"Ar`��   Ar`�P   ArcP   ?��Arc3�   ArcP   AreP   ?��Are��   AreP   Arg�P   ?��Arh�   Arg�P   ArjaP   ?���Arj��   ArjaP   Arl�P   ?iArl��   Arl�P   AroCP   ?��Aroh�   AroCP   Arq�P   ?��Arq��   Arq�P   Art%P   ?�0�ArtJ�   Art%P   Arv�P   ?�ݰArv��   Arv�P   AryP   ?r}dAry,�   AryP   Ar{xP   ?0�_Ar{��   Ar{xP   Ar}�P   ?�`�Ar~�   Ar}�P   Ar�ZP   ?�ZAr��   Ar�ZP   Ar��P   ?�`CAr���   Ar��P   Ar�<P   ?}��Ar�a�   Ar�<P   Ar��P   ?�́Ar���   Ar��P   Ar�P   ?��kAr�C�   Ar�P   Ar��P   ?�5�Ar���   Ar��P   Ar� P   ?���Ar�%�   Ar� P   Ar�qP   ?�R�Ar���   Ar�qP   Ar��P   ?ا�Ar��   Ar��P   Ar�SP   ?�Ar�x�   Ar�SP   Ar��P   ?�e�Ar���   Ar��P   Ar�5P   ?���Ar�Z�   Ar�5P   Ar��P   ?:�KAr���   Ar��P   Ar�P   ?���Ar�<�   Ar�P   Ar��P   ?��Ar���   Ar��P   Ar��P   ?�bAr��   Ar��P   Ar�jP   ?�#`Ar���   Ar�jP   Ar��P   ?��Ar� �   Ar��P   Ar�LP   ?�W�Ar�q�   Ar�LP   Ar��P   ?��iAr���   Ar��P   Ar�.P   ?��fAr�S�   Ar�.P   Ar��P   ?���Ar���   Ar��P   Ar�P   ?��Ar�5�   Ar�P   Ar��P   ?���Ar���   Ar��P   Ar��P   �HN`Ar��   Ar��P   Ar�cP   ?G1Ar���   Ar�cP   Ar��P   ?h)aAr���   Ar��P   Ar�EP   ?��qAr�j�   Ar�EP   ArĶP   ?V�|Ar���   ArĶP   Ar�'P   ?�\�Ar�L�   Ar�'P   ArɘP   @	qArɽ�   ArɘP   Ar�	P   ?�ɄAr�.�   Ar�	P   Ar�zP   ?��fArΟ�   Ar�zP   Ar��P   ?�2�Ar��   Ar��P   Ar�\P   ?�m�ArӁ�   Ar�\P   Ar��P   ?W�5Ar���   Ar��P   Ar�>P   ?q��Ar�c�   Ar�>P   ArگP   ?�yAr���   ArگP   Ar� P   ?���Ar�E�   Ar� P   ArߑP   ?ҫ	Ar߶�   ArߑP   Ar�P   ?Ȗ�Ar�'�   Ar�P   Ar�sP   ?�oAr��   Ar�sP   Ar��P   ?�q+Ar�	�   Ar��P   Ar�UP   ?�Ar�z�   Ar�UP   Ar��P   ?��Ar���   Ar��P   Ar�7P   ?��Ar�\�   Ar�7P   Ar�P   ?�̘Ar���   Ar�P   Ar�P   ?�pAr�>�   Ar�P   Ar��P   ?��Ar���   Ar��P   Ar��P   ?�|Ar� �   Ar��P   Ar�lP   ?��Ar���   Ar�lP   Ar��P   @�Ar��   Ar��P   Ar�NP   >��Ar�s�   Ar�NP   As�P   �`�EAs��   As�P   As0P   ?��AsU�   As0P   As�P   ?�B�As��   As�P   As	P   ?�0�As	7�   As	P   As�P   @�As��   As�P   As�P   @v�As�   As�P   AseP   ?�E�As��   AseP   As�P   ?��As��   As�P   AsGP   ?���Asl�   AsGP   As�P   @PAs��   As�P   As)P   @ǗAsN�   As)P   As�P   ?{�:As��   As�P   AsP   ?�t{As0�   AsP   As!|P   @�YAs!��   As!|P   As#�P   ?鈺As$�   As#�P   As&^P   ?��-As&��   As&^P   As(�P   ?���As(��   As(�P   As+@P   ?�As+e�   As+@P   As-�P   ?�wAs-��   As-�P   As0"P   ?�#�As0G�   As0"P   As2�P   ?�]As2��   As2�P   As5P   ?�RuAs5)�   As5P   As7uP   ?���